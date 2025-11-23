# Quick Start - Nova Arquitetura Modular

Guia rápido para começar a usar a nova arquitetura modular do RITM Annotation.

## Instalação

```bash
# Clone o repositório
git clone https://github.com/lucasew/ritm_annotation.git
cd ritm_annotation

# Instale dependências
pip install -e .

# (Opcional) Para API Web
pip install fastapi uvicorn python-multipart
```

## Uso Básico

### 1. Anotação Interativa (GUI)

A GUI atual continua funcionando normalmente:

```bash
python -m ritm_annotation annotate \
    --input_path images/ \
    --output_path annotations/ \
    --classes person car tree \
    --checkpoint models/checkpoint.pth
```

### 2. Anotação Programática

Use a nova API para anotação sem GUI:

```python
from pathlib import Path
import cv2
from ritm_annotation.core.annotation import AnnotationSession
from ritm_annotation.inference.utils import load_is_model
from ritm_annotation.inference.predictors import get_predictor

# Carregar modelo
model = load_is_model("checkpoint.pth", device="cuda")
predictor = get_predictor(model, device="cuda")

# Criar sessão
session = AnnotationSession(predictor, prob_thresh=0.5)

# Carregar e anotar imagem
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
session.load_image(image)

# Adicionar clicks
session.add_click(100, 150, is_positive=True)   # Click positivo
session.add_click(200, 250, is_positive=True)
session.add_click(180, 200, is_positive=False)  # Click negativo

# Obter resultado
mask = session.get_current_prediction()

# Finalizar objeto
session.finish_object()
final_mask = session.get_result_mask()

# Salvar
cv2.imwrite("mask.png", (mask * 255).astype('uint8'))
```

### 3. Treinamento Modular

Use os novos componentes de treinamento:

```python
from pathlib import Path
from ritm_annotation.core.training import (
    TrainingLoop,
    BatchProcessor,
    CheckpointManager,
)
from torch.utils.data import DataLoader

# Preparar dados
train_dataset = MyDataset(...)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Configurar componentes
batch_processor = BatchProcessor(
    model=model,
    loss_fn=loss_config,
    metrics=[iou_metric],
    max_interactive_points=3,
)

checkpoint_manager = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    model_name="my_model",
    keep_best_n=3,
    save_every_n_epochs=10,
)

# Criar training loop
loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_manager=checkpoint_manager,
)

# Treinar
history = loop.run(num_epochs=100)
```

### 4. API Web (Experimental)

Execute a API web de exemplo:

```bash
# Instalar dependências
pip install fastapi uvicorn python-multipart

# Executar servidor
python examples/web_api_example.py

# Acesse http://localhost:8000/docs para ver a documentação da API
```

Exemplo de uso da API:

```python
import requests

# Criar sessão
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/session/create",
        files={"image": f}
    )
session_id = response.json()["session_id"]

# Adicionar click
response = requests.post(
    f"http://localhost:8000/session/{session_id}/click",
    json={"x": 100, "y": 150, "is_positive": True}
)

# Obter resultado
response = requests.get(
    f"http://localhost:8000/session/{session_id}/result"
)
mask_base64 = response.json()["mask_base64"]
```

## Estrutura da Nova Arquitetura

```
ritm_annotation/
├── core/                    # Lógica core (UI-agnostic)
│   ├── annotation/         # Sistema de anotação
│   │   ├── session.py      # AnnotationSession
│   │   ├── state.py        # Classes de estado
│   │   └── events.py       # Sistema de eventos
│   └── training/           # Sistema de treinamento
│       ├── loop.py         # TrainingLoop
│       ├── batch_processor.py
│       ├── checkpoint_manager.py
│       └── metrics_tracker.py
├── interfaces/             # Adaptadores de UI
│   └── gui_adapter.py      # Adaptador para Tkinter
└── examples/               # Exemplos de uso
    └── web_api_example.py
```

## Principais Componentes

### AnnotationSession

Gerencia o estado da anotação interativa:

```python
session = AnnotationSession(predictor, prob_thresh=0.5)

# Carregar imagem
session.load_image(image, image_path="image.jpg")

# Interagir
session.add_click(x, y, is_positive=True)
session.undo_click()
session.reset_clicks()
session.finish_object()

# Obter dados
state = session.state
viz_data = session.get_visualization_data()
mask = session.get_result_mask()
```

### TrainingLoop

Orquestra o treinamento:

```python
loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    train_loader=train_loader,
    val_loader=val_loader,
)

# Treinar
metrics = loop.run(num_epochs=100)

# Retomar de checkpoint
start_epoch = loop.resume_from_checkpoint("checkpoint.pth")
loop.run(num_epochs=100, start_epoch=start_epoch)
```

### BatchProcessor

Processa batches de treino:

```python
processor = BatchProcessor(
    model=model,
    loss_fn=loss_config,
    metrics=[metric1, metric2],
    max_interactive_points=3,
)

# Durante treino
loss, loss_dict, batch_data, outputs = processor.process_batch(
    batch,
    device='cuda',
    is_training=True
)
```

## Sistema de Eventos

Subscribe a eventos da sessão:

```python
from ritm_annotation.core.annotation import EventType

def on_click_added(event):
    print(f"Click adicionado: {event.data}")

def on_prediction_complete(event):
    print("Predição concluída!")

session.events.on(EventType.CLICK_ADDED, on_click_added)
session.events.on(EventType.PREDICTION_COMPLETED, on_prediction_complete)
```

## Exemplos Avançados

### Processar batch de imagens

```python
from pathlib import Path

def annotate_batch(image_dir, output_dir, model_path):
    """Anotar múltiplas imagens com clicks pré-definidos."""

    # Setup
    model = load_is_model(model_path, device="cuda")
    predictor = get_predictor(model, device="cuda")
    session = AnnotationSession(predictor)

    # Processar cada imagem
    for img_path in Path(image_dir).glob("*.jpg"):
        # Carregar imagem
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        session.load_image(image)

        # Carregar clicks (de arquivo JSON, CSV, etc)
        clicks = load_clicks_for_image(img_path)

        # Aplicar clicks
        for click in clicks:
            session.add_click(
                click['x'],
                click['y'],
                is_positive=click['positive']
            )

        # Finalizar e salvar
        session.finish_object()
        mask = session.get_result_mask()

        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), mask)

        print(f"✓ {img_path.name}")
```

### Treinamento com callbacks

```python
def train_with_logging(model, train_loader, val_loader):
    """Treinar com logging customizado."""

    # Setup
    loop = TrainingLoop(...)

    # Definir callbacks
    def on_epoch_end(epoch, metrics):
        # Log metrics
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {metrics.train_loss:.4f}")
        print(f"  Val Loss: {metrics.val_loss:.4f}")

        # Salvar gráficos
        plot_metrics(metrics)

        # Early stopping
        if should_stop(metrics):
            return False  # Para o treinamento

        return True  # Continua

    # Executar
    callbacks = {'on_epoch_end': on_epoch_end}
    loop.run(num_epochs=100, callbacks=callbacks)
```

## Comparação: Antes vs Depois

### Anotação

**Antes:**
```python
# Acoplado à GUI
controller = InteractiveController(model, device, update_image_callback=...)
controller.set_image(image)
controller.add_click(x, y, True)
```

**Depois:**
```python
# Desacoplado, pode usar em qualquer contexto
session = AnnotationSession(predictor)
session.load_image(image)
session.add_click(x, y, True)

# Para GUI, use adapter
adapter = GUIAdapter(session, update_image_callback=...)
```

### Treinamento

**Antes:**
```python
# Monolítico
trainer = ISTrainer(model, cfg, model_cfg, loss_cfg, trainset, valset, ...)
trainer.run(num_epochs)
```

**Depois:**
```python
# Modular
processor = BatchProcessor(model, loss_fn, metrics)
manager = CheckpointManager(checkpoint_dir)
loop = TrainingLoop(model, optimizer, processor, train_loader, val_loader, manager)
loop.run(num_epochs)
```

## Próximos Passos

1. **Explore a documentação**:
   - `docs/ARCHITECTURE.md` - Arquitetura detalhada
   - `docs/MIGRATION_GUIDE.md` - Guia de migração

2. **Experimente os exemplos**:
   - `examples/web_api_example.py` - API Web
   - Veja testes para mais exemplos

3. **Crie sua interface**:
   - Implemente um adapter para sua UI
   - Reutilize `AnnotationSession` e `TrainingLoop`

## Recursos

- **Documentação**: `docs/`
- **Exemplos**: `examples/`
- **Testes**: `tests/`
- **Issues**: GitHub Issues

## Suporte

Para dúvidas ou problemas:
1. Consulte a documentação em `docs/`
2. Veja os exemplos em `examples/`
3. Abra uma issue no GitHub

## Contribuindo

Contribuições são bem-vindas! A arquitetura modular facilita adicionar novas funcionalidades:

1. **Nova interface**: Crie um adapter em `interfaces/`
2. **Nova feature de treino**: Estenda componentes em `core/training/`
3. **Nova métrica**: Adicione em `model/metrics.py`
4. **Novo tipo de dado**: Adicione em `data/datasets/`

Veja `CONTRIBUTING.md` para mais detalhes.
