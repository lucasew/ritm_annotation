# Arquitetura Modular - RITM Annotation

## Visão Geral

Este documento descreve a nova arquitetura modular do sistema de anotação e treinamento RITM. O objetivo é criar um sistema mais limpo, extensível e fácil de manter.

## Princípios de Design

1. **Separação de Responsabilidades**: Lógica de negócio separada da interface do usuário
2. **Modularidade**: Componentes pequenos e focados que podem ser compostos
3. **Extensibilidade**: Fácil adicionar novas funcionalidades sem modificar código existente
4. **Testabilidade**: Componentes isolados são mais fáceis de testar
5. **Reutilização**: Mesma lógica core pode ser usada em múltiplas interfaces

## Estrutura de Diretórios

```
ritm_annotation/
├── core/                       # Lógica de negócio (sem dependências de UI)
│   ├── annotation/            # Sistema de anotação
│   │   ├── session.py        # AnnotationSession - gerenciamento de estado
│   │   ├── state.py          # Classes de estado (Click, ObjectState, etc)
│   │   └── events.py         # Sistema de eventos pub/sub
│   │
│   └── training/              # Sistema de treinamento
│       ├── loop.py           # TrainingLoop - orquestração de treino
│       ├── batch_processor.py # BatchProcessor - processamento de batches
│       ├── checkpoint_manager.py # Gerenciamento de checkpoints
│       └── metrics_tracker.py    # Rastreamento de métricas
│
├── interfaces/                # Adaptadores de UI
│   ├── gui_adapter.py        # Adaptador para Tkinter GUI
│   └── web_adapter.py        # Adaptador para Web API (futuro)
│
├── cli/                       # Interface de linha de comando
│   ├── annotate/             # Comando de anotação
│   ├── train/                # Comando de treinamento
│   └── ...
│
├── model/                     # Modelos de rede neural
├── inference/                 # Lógica de inferência
├── data/                      # Datasets e transformações
└── engine/                    # Engine de treinamento (legacy)
```

## Componentes Core

### 1. Sistema de Anotação (`core/annotation/`)

#### AnnotationSession

Classe central que gerencia uma sessão de anotação interativa.

**Responsabilidades:**
- Gerenciamento de estado da anotação
- Coordenação com o predictor (modelo)
- Histórico de ações para undo/redo
- Emissão de eventos para UI

**Principais Métodos:**
```python
session = AnnotationSession(predictor, prob_thresh=0.5)

# Carregar imagem
session.load_image(image, image_path)

# Adicionar clicks
prob_map = session.add_click(x, y, is_positive=True)

# Desfazer último click
session.undo_click()

# Finalizar objeto atual
session.finish_object()

# Obter dados para visualização
viz_data = session.get_visualization_data()
```

**Vantagens:**
- UI-agnostic: pode ser usado com qualquer interface
- Testável: sem dependências de GUI
- Reutilizável: mesma lógica para GUI, Web, CLI

#### Sistema de Eventos

Padrão pub/sub para desacoplar lógica de UI.

```python
# Definir callback
def on_prediction_complete(event: AnnotationEvent):
    print(f"Prediction completed: {event.data}")

# Registrar listener
session.events.on(EventType.PREDICTION_COMPLETED, on_prediction_complete)

# Eventos são emitidos automaticamente pela session
```

**Tipos de Eventos:**
- `IMAGE_LOADED`: Imagem carregada
- `CLICK_ADDED`: Click adicionado
- `PREDICTION_COMPLETED`: Predição concluída
- `OBJECT_FINISHED`: Objeto finalizado
- `MASK_LOADED`: Máscara carregada

#### Classes de Estado

Representam o estado da anotação de forma serializable:

```python
@dataclass
class Click:
    x: float
    y: float
    is_positive: bool
    object_id: int

@dataclass
class ObjectState:
    object_id: int
    clicks: List[Click]
    mask: Optional[np.ndarray]
    is_finished: bool

@dataclass
class AnnotationState:
    image_path: Optional[str]
    current_object_id: int
    objects: List[ObjectState]
    result_mask: Optional[np.ndarray]
```

### 2. Sistema de Treinamento (`core/training/`)

#### TrainingLoop

Orquestra o loop de treinamento de forma modular.

**Responsabilidades:**
- Executar epochs de treino e validação
- Coordenar batch processor, métricas, checkpoints
- Gerenciar scheduler de learning rate
- Callbacks para extensibilidade

**Uso:**
```python
loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=scheduler,
    checkpoint_manager=checkpoint_manager,
)

# Executar treinamento
metrics = loop.run(num_epochs=100)
```

#### BatchProcessor

Encapsula a lógica de processar um batch.

**Responsabilidades:**
- Forward pass com simulação de clicks interativos
- Cálculo de loss
- Atualização de métricas

**Componentes:**
- Simulação de clicks iterativos durante treino
- Computação de losses (instance, auxiliary)
- Atualização de métricas

```python
processor = BatchProcessor(
    model=model,
    loss_fn=loss_config,
    metrics=[iou_metric],
    max_interactive_points=3,
)

loss, loss_dict, batch_data, outputs = processor.process_batch(
    batch, device, is_training=True
)
```

#### CheckpointManager

Gerencia salvamento e carregamento de checkpoints.

**Features:**
- Salvamento periódico
- Manter N melhores checkpoints
- Salvamento de estado completo (model, optimizer, scheduler)
- Metadados (epoch, metrics)

```python
manager = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    keep_best_n=3,
    save_every_n_epochs=10,
)

# Salvar checkpoint
manager.save_checkpoint(
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    metrics=metrics,
)

# Carregar checkpoint
manager.load_checkpoint(path, model, optimizer)
```

#### MetricsTracker

Rastreia e agrega métricas durante treinamento.

**Features:**
- Acumulação de métricas por batch
- Cálculo de médias por epoch
- Histórico de métricas
- Rastreamento de melhores valores

```python
tracker = MetricsTracker()

# Durante treino
for batch in dataloader:
    loss, metrics = process_batch(batch)
    tracker.update(metrics)

# Fim do epoch
epoch_metrics = tracker.end_epoch()
```

## Interfaces

### GUI Adapter

Adaptador que conecta `AnnotationSession` à GUI Tkinter existente.

**Responsabilidades:**
- Traduzir eventos da session em callbacks da GUI
- Fornecer métodos compatíveis com o controller antigo
- Renderizar visualizações

```python
adapter = GUIAnnotationAdapter(
    session=session,
    update_image_callback=self.update_image,
)

# Uso compatível com código antigo
adapter.set_image(image)
adapter.add_click(x, y, is_positive=True)
vis = adapter.get_visualization()
```

### Web API (Exemplo)

Demonstra como criar uma API Web usando a mesma lógica core.

**Endpoints:**
- `POST /session/create`: Criar nova sessão de anotação
- `POST /session/{id}/click`: Adicionar click
- `POST /session/{id}/undo`: Desfazer click
- `GET /session/{id}/result`: Obter máscara final

Ver `examples/web_api_example.py` para implementação completa.

## Fluxo de Dados

### Anotação

```
User Action (GUI/Web)
    ↓
Interface Adapter
    ↓
AnnotationSession.add_click()
    ↓
Predictor.get_prediction()
    ↓
Model Inference
    ↓
Update State
    ↓
Emit Event
    ↓
Interface Callback
    ↓
Update UI
```

### Treinamento

```
TrainingLoop.run()
    ↓
For each epoch:
    ↓
    Train Epoch
        ↓
        For each batch:
            ↓
            BatchProcessor.process_batch()
                ↓
                Simulate clicks
                ↓
                Forward pass
                ↓
                Compute loss
                ↓
                Update metrics
            ↓
            Backward + Optimize
        ↓
        MetricsTracker.end_epoch()
    ↓
    Validation Epoch (similar)
    ↓
    Scheduler.step()
    ↓
    CheckpointManager.save()
```

## Benefícios da Nova Arquitetura

### 1. **Separação de Responsabilidades**

**Antes:**
- `InteractiveController`: Mistura lógica de negócio com callbacks de GUI
- `ISTrainer`: Classe monolítica com muitas responsabilidades

**Depois:**
- `AnnotationSession`: Apenas lógica de anotação
- `GUIAdapter`: Apenas interface com GUI
- `TrainingLoop`, `BatchProcessor`, etc: Componentes focados

### 2. **Extensibilidade**

**Adicionar nova interface é fácil:**

```python
# Criar novo adaptador
class WebAPIAdapter:
    def __init__(self, session: AnnotationSession):
        self.session = session
        # Subscribe to events
        session.events.on(EventType.PREDICTION_COMPLETED, self.on_prediction)

    def on_prediction(self, event):
        # Send update via WebSocket, etc
        pass
```

**Adicionar nova funcionalidade de treino:**

```python
# Criar novo componente
class DistributedCheckpointManager(CheckpointManager):
    def save_checkpoint(self, ...):
        # Save to distributed storage
        pass

# Usar no TrainingLoop
loop = TrainingLoop(..., checkpoint_manager=DistributedCheckpointManager(...))
```

### 3. **Testabilidade**

```python
# Testar lógica de anotação sem GUI
def test_annotation_session():
    predictor = MockPredictor()
    session = AnnotationSession(predictor)

    session.load_image(test_image)
    session.add_click(100, 100, is_positive=True)

    assert len(session.state.get_current_object().clicks) == 1
```

### 4. **Reutilização**

Mesma `AnnotationSession` pode ser usada em:
- GUI Desktop (Tkinter)
- Web API (FastAPI)
- CLI (interativo)
- Jupyter Notebook
- Mobile (PyQt/Kivy)

## Migração Gradual

A nova arquitetura é compatível com o código antigo:

1. **GUI atual**: Usa `GUIAdapter` que mantém interface compatível
2. **Comandos CLI**: Podem usar novos componentes gradualmente
3. **Legacy code**: Ainda funciona, pode ser migrado aos poucos

## Próximos Passos

1. **Completar migração da GUI**: Substituir `InteractiveController` por `GUIAdapter`
2. **Migrar comandos de treino**: Usar `TrainingLoop` no lugar de `ISTrainer`
3. **Adicionar testes**: Testar componentes core isoladamente
4. **Web interface**: Implementar interface web completa
5. **API documentation**: Documentar APIs dos componentes core
6. **Performance profiling**: Otimizar componentes críticos

## Exemplos de Uso

### Criar sessão de anotação programática

```python
from ritm_annotation.core.annotation import AnnotationSession
from ritm_annotation.inference.utils import load_is_model
from ritm_annotation.inference.predictors import get_predictor

# Carregar modelo
model = load_is_model("checkpoint.pth", device="cuda")
predictor = get_predictor(model, device="cuda")

# Criar sessão
session = AnnotationSession(predictor, prob_thresh=0.5)

# Anotar imagem
session.load_image(image)
session.add_click(100, 150, is_positive=True)
session.add_click(200, 250, is_positive=True)
session.add_click(180, 200, is_positive=False)

# Finalizar e obter resultado
session.finish_object()
mask = session.get_result_mask()
```

### Treinar modelo com novos componentes

```python
from ritm_annotation.core.training import (
    TrainingLoop, BatchProcessor, CheckpointManager
)

# Setup componentes
batch_processor = BatchProcessor(
    model=model,
    loss_fn=loss_config,
    metrics=[iou_metric],
)

checkpoint_manager = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    keep_best_n=3,
)

loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_manager=checkpoint_manager,
)

# Treinar
metrics = loop.run(num_epochs=100)
```

## Conclusão

A nova arquitetura modular torna o RITM Annotation mais:
- **Maintainable**: Código organizado e fácil de entender
- **Extensible**: Fácil adicionar novas features
- **Testable**: Componentes isolados
- **Reusable**: Mesma lógica em múltiplas interfaces
- **Flexible**: Componentes podem ser compostos de diferentes formas

Tudo isso mantendo compatibilidade com código existente e permitindo migração gradual.
