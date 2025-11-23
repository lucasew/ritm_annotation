

# Guia de Migração - Nova Arquitetura Modular

Este guia ajuda a migrar código existente para usar a nova arquitetura modular.

## Visão Geral

A nova arquitetura separa:
- **Core Logic** (`core/`): Lógica de negócio sem dependências de UI
- **Interfaces** (`interfaces/`): Adaptadores para diferentes UIs
- **CLI** (`cli/`): Comandos de linha de comando

## Migração do Sistema de Anotação

### Antes (InteractiveController)

```python
from ritm_annotation.cli.annotate.controller import InteractiveController

# Criar controller
controller = InteractiveController(
    model, device,
    predictor_params=predictor_params,
    update_image_callback=self.update_image,
)

# Usar controller
controller.set_image(image)
controller.add_click(x, y, is_positive=True)
controller.undo_click()
controller.partially_finish_object()
vis = controller.get_visualization(alpha_blend, click_radius)
```

### Depois (AnnotationSession + Adapter)

```python
from ritm_annotation.core.annotation import AnnotationSession
from ritm_annotation.interfaces import GUIAnnotationAdapter
from ritm_annotation.inference.predictors import get_predictor

# Criar predictor
predictor = get_predictor(
    model,
    device=device,
    **predictor_params
)

# Criar sessão
session = AnnotationSession(
    predictor=predictor,
    prob_thresh=0.5,
)

# Criar adaptador para GUI
adapter = GUIAnnotationAdapter(
    session=session,
    update_image_callback=self.update_image,
    click_radius=click_radius,
)

# Usar (API idêntica ao controller antigo)
adapter.set_image(image)
adapter.add_click(x, y, is_positive=True)
adapter.undo_click()
adapter.partially_finish_object()
vis = adapter.get_visualization(alpha_blend, click_radius)
```

### Benefícios

1. **Mesma API**: Código da GUI não precisa mudar
2. **Mais flexível**: Pode trocar adapter sem mudar core
3. **Testável**: Session pode ser testada sem GUI
4. **Extensível**: Fácil criar adapter para Web, CLI, etc

## Migração do Sistema de Treinamento

### Antes (ISTrainer monolítico)

```python
from ritm_annotation.engine.trainer import ISTrainer

trainer = ISTrainer(
    model=model,
    cfg=cfg,
    model_cfg=model_cfg,
    loss_cfg=loss_cfg,
    trainset=trainset,
    valset=valset,
    optimizer='adam',
    lr=5e-4,
    checkpoint_interval=10,
    # ... muitos outros parâmetros
)

trainer.run(num_epochs=230)
```

### Depois (Componentes modulares)

```python
from ritm_annotation.core.training import (
    TrainingLoop,
    BatchProcessor,
    CheckpointManager,
    MetricsTracker,
)

# 1. Criar batch processor
batch_processor = BatchProcessor(
    model=model,
    loss_fn=loss_cfg,
    metrics=[iou_metric],
    max_interactive_points=model_cfg.get('num_max_points', 24),
)

# 2. Criar checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir=cfg.CHECKPOINTS_PATH,
    model_name=cfg.exp_name,
    keep_best_n=3,
    save_every_n_epochs=10,
)

# 3. Criar data loaders
train_loader = DataLoader(trainset, batch_size=cfg.batch_size, ...)
val_loader = DataLoader(valset, batch_size=cfg.batch_size, ...)

# 4. Criar training loop
loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=scheduler,
    checkpoint_manager=checkpoint_manager,
    device=cfg.device,
)

# 5. Executar
metrics = loop.run(num_epochs=230)
```

### Benefícios

1. **Modular**: Cada componente faz uma coisa
2. **Flexível**: Fácil trocar componentes
3. **Reutilizável**: Componentes podem ser usados em outros contextos
4. **Testável**: Cada componente pode ser testado isoladamente

## Padrões Comuns de Migração

### 1. Callbacks para Eventos

**Antes:**
```python
def update_callback():
    # Update UI
    pass

controller = InteractiveController(
    model, device,
    update_image_callback=update_callback
)
```

**Depois:**
```python
from ritm_annotation.core.annotation import EventType

# Subscribe to events
def on_prediction(event):
    # Update UI
    pass

session.events.on(EventType.PREDICTION_COMPLETED, on_prediction)
```

### 2. State Management

**Antes:**
```python
# Estado era gerenciado internamente
states = controller.states
probs_history = controller.probs_history
```

**Depois:**
```python
# Estado é explícito e acessível
state = session.state
current_object = state.get_current_object()
clicks = current_object.clicks
mask = current_object.mask

# Histórico para undo é gerenciado internamente
session.undo_click()  # Automaticamente restaura estado anterior
```

### 3. Visualização

**Antes:**
```python
# Visualização era responsabilidade do controller
vis = controller.get_visualization(alpha_blend=0.5)
```

**Depois:**
```python
# Adapter fornece visualização (compatível)
vis = adapter.get_visualization(alpha_blend=0.5)

# Ou usar dados brutos da session
viz_data = session.get_visualization_data()
# viz_data contém: image, masks, clicks, etc
# Você pode criar sua própria visualização
```

### 4. Métricas de Treinamento

**Antes:**
```python
# ISTrainer gerenciava métricas internamente
# Difícil acessar ou customizar
```

**Depois:**
```python
# Metrics tracker explícito
tracker = MetricsTracker()

# Durante treino
for batch in dataloader:
    loss, metrics = process_batch(batch)
    tracker.update(metrics)

# Fim do epoch
epoch_metrics = tracker.end_epoch()
print(f"IoU: {epoch_metrics['iou']:.4f}")

# Histórico completo
history = tracker.get_history()
```

## Exemplos Práticos

### Exemplo 1: Anotação Programática

```python
"""Anotar imagens em batch sem GUI."""

from pathlib import Path
import cv2
from ritm_annotation.core.annotation import AnnotationSession
from ritm_annotation.inference.utils import load_is_model
from ritm_annotation.inference.predictors import get_predictor

# Setup
model = load_is_model("checkpoint.pth", device="cuda")
predictor = get_predictor(model, device="cuda")
session = AnnotationSession(predictor)

# Processar imagens
input_dir = Path("images")
output_dir = Path("masks")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    # Carregar imagem
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    session.load_image(image, str(img_path))

    # Adicionar clicks (pode ser de arquivo de configuração)
    clicks = load_clicks_from_file(img_path.with_suffix(".json"))
    for click in clicks:
        session.add_click(click['x'], click['y'], click['is_positive'])

    # Finalizar e salvar
    session.finish_object()
    mask = session.get_result_mask()

    # Salvar máscara
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), mask)

    print(f"Processed {img_path.name}")
```

### Exemplo 2: Treinamento Customizado

```python
"""Treinamento com logging customizado e early stopping."""

from ritm_annotation.core.training import TrainingLoop
import wandb

# Setup componentes
loop = TrainingLoop(...)

# Callbacks customizados
def on_epoch_end(epoch, metrics):
    # Log para Weights & Biases
    wandb.log({
        'epoch': epoch,
        'train_loss': metrics.train_loss,
        'val_loss': metrics.val_loss,
        **metrics.val_metrics,
    })

    # Early stopping
    if metrics.val_loss < best_loss:
        best_loss = metrics.val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            return False  # Stop training

    return True  # Continue training

# Executar com callbacks
callbacks = {'on_epoch_end': on_epoch_end}
loop.run(num_epochs=1000, callbacks=callbacks)
```

### Exemplo 3: Web API

```python
"""API Web para anotação interativa."""

from fastapi import FastAPI, UploadFile
from ritm_annotation.core.annotation import AnnotationSession

app = FastAPI()
sessions = {}  # session_id -> AnnotationSession

@app.post("/session/create")
async def create_session(image: UploadFile):
    # Criar nova sessão
    session = AnnotationSession(predictor)
    image_data = decode_image(await image.read())
    session.load_image(image_data)

    # Armazenar
    session_id = generate_id()
    sessions[session_id] = session

    return {"session_id": session_id}

@app.post("/session/{session_id}/click")
async def add_click(session_id: str, x: float, y: float, is_positive: bool):
    session = sessions[session_id]
    prob_map = session.add_click(x, y, is_positive)

    return {
        "mask": encode_mask(prob_map),
        "num_clicks": len(session.state.get_current_object().clicks)
    }

# Ver examples/web_api_example.py para implementação completa
```

### Exemplo 4: Teste Unitário

```python
"""Testes para lógica de anotação."""

import pytest
import numpy as np
from ritm_annotation.core.annotation import AnnotationSession

class MockPredictor:
    """Predictor simulado para testes."""

    def set_input_image(self, image):
        self.image = image

    def get_prediction(self, clicker):
        # Retorna máscara circular ao redor do primeiro click
        clicks = clicker.get_clicks()
        if not clicks:
            return np.zeros((100, 100))

        mask = np.zeros((100, 100))
        x, y = int(clicks[0].coords[0]), int(clicks[0].coords[1])
        cv2.circle(mask, (x, y), 20, 1.0, -1)
        return mask

def test_annotation_session():
    # Criar sessão com predictor simulado
    predictor = MockPredictor()
    session = AnnotationSession(predictor)

    # Carregar imagem
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    session.load_image(image)

    # Adicionar click
    session.add_click(50, 50, is_positive=True)

    # Verificar estado
    assert len(session.state.get_current_object().clicks) == 1

    # Verificar predição
    prob_map = session.get_current_prediction()
    assert prob_map is not None
    assert prob_map.shape == (100, 100)

    # Testar undo
    session.undo_click()
    assert len(session.state.get_current_object().clicks) == 0

def test_multi_object_annotation():
    predictor = MockPredictor()
    session = AnnotationSession(predictor)
    session.load_image(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    # Primeiro objeto
    session.add_click(30, 30, is_positive=True)
    session.finish_object()

    # Segundo objeto
    session.add_click(70, 70, is_positive=True)
    session.finish_object()

    # Verificar resultado
    result_mask = session.get_result_mask()
    assert result_mask is not None
    assert len(np.unique(result_mask)) == 3  # Background + 2 objects
```

## Checklist de Migração

### Para Anotação

- [ ] Substituir `InteractiveController` por `AnnotationSession` + `GUIAdapter`
- [ ] Migrar callbacks para event listeners
- [ ] Atualizar testes para usar session diretamente
- [ ] Verificar compatibilidade de visualização

### Para Treinamento

- [ ] Substituir `ISTrainer` por `TrainingLoop` + componentes
- [ ] Migrar configuração de losses para `BatchProcessor`
- [ ] Configurar `CheckpointManager` com parâmetros apropriados
- [ ] Adicionar logging/callbacks customizados se necessário
- [ ] Atualizar scripts de treinamento

### Geral

- [ ] Atualizar documentação
- [ ] Adicionar testes unitários para novos componentes
- [ ] Verificar performance (não deve degradar)
- [ ] Atualizar exemplos e tutoriais

## Perguntas Frequentes

### Q: O código antigo ainda funciona?

**A:** Sim! A nova arquitetura é compatível. `GUIAdapter` mantém a mesma interface que `InteractiveController`.

### Q: Preciso migrar tudo de uma vez?

**A:** Não! Você pode migrar gradualmente:
1. Comece usando adapters com código existente
2. Migre partes específicas conforme necessário
3. Adicione novas features usando componentes novos

### Q: Como faço para...

**...adicionar uma nova interface?**
```python
# Crie um novo adapter
class MyAdapter:
    def __init__(self, session: AnnotationSession):
        self.session = session
        # Subscribe to events
        session.events.on(EventType.PREDICTION_COMPLETED, self.on_update)
```

**...customizar o treinamento?**
```python
# Crie componentes customizados
class MyBatchProcessor(BatchProcessor):
    def process_batch(self, batch, device, is_training):
        # Custom logic
        pass

# Use no TrainingLoop
loop = TrainingLoop(..., batch_processor=MyBatchProcessor(...))
```

**...acessar o estado interno?**
```python
# Session expõe estado explicitamente
state = session.state
current_object = state.get_current_object()
clicks = current_object.clicks
mask = current_object.mask
```

## Suporte

Para dúvidas ou problemas:
1. Consulte `docs/ARCHITECTURE.md` para entender a arquitetura
2. Veja exemplos em `examples/`
3. Procure testes em `tests/` para ver uso prático
4. Abra uma issue no GitHub

## Conclusão

A migração é gradual e não-destrutiva. Comece usando adapters para manter compatibilidade, depois migre partes específicas conforme necessário. A nova arquitetura oferece muito mais flexibilidade e extensibilidade.
