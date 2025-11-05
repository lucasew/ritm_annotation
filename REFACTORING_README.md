# 🎯 Reestruturação Modular - RITM Annotation

## 📋 Resumo

Este projeto foi reestruturado para uma arquitetura mais modular, extensível e fácil de manter. A principal mudança é a **separação da lógica de negócio das interfaces de usuário**, permitindo usar o mesmo código core em diferentes contextos (GUI, Web, CLI, etc).

## 🎁 O que há de novo

### ✨ Sistema de Anotação Modular

- **`AnnotationSession`**: Gerencia anotação interativa de forma independente de UI
- **Sistema de Eventos**: Arquitetura pub/sub para desacoplar componentes
- **State Management**: Estado explícito e serializable
- **Multi-interface**: Mesma lógica funciona em GUI, Web, CLI

### 🎓 Sistema de Treinamento Modular

- **`TrainingLoop`**: Loop de treinamento modular e customizável
- **`BatchProcessor`**: Lógica de processamento de batch isolada
- **`CheckpointManager`**: Gerenciamento profissional de checkpoints
- **`MetricsTracker`**: Rastreamento de métricas com histórico

### 🔌 Interfaces Desacopladas

- **`GUIAdapter`**: Conecta core à GUI Tkinter (compatível com código existente)
- **Web API**: Exemplo de API REST com FastAPI
- **Extensível**: Fácil criar adapters para outras interfaces

## 📁 Nova Estrutura

```
ritm_annotation/
├── core/                          # 🎯 Lógica core (sem dependências de UI)
│   ├── annotation/               # Sistema de anotação
│   │   ├── session.py           # AnnotationSession - gerencia estado
│   │   ├── state.py             # Classes de estado (Click, ObjectState, etc)
│   │   └── events.py            # Sistema de eventos pub/sub
│   │
│   └── training/                 # Sistema de treinamento
│       ├── loop.py              # TrainingLoop - orquestra treino
│       ├── batch_processor.py   # Processa batches
│       ├── checkpoint_manager.py # Gerencia checkpoints
│       └── metrics_tracker.py    # Rastreia métricas
│
├── interfaces/                    # 🔌 Adaptadores de UI
│   └── gui_adapter.py            # Adaptador para Tkinter GUI
│
├── examples/                      # 📚 Exemplos de uso
│   └── web_api_example.py        # API Web com FastAPI
│
└── docs/                          # 📖 Documentação
    ├── ARCHITECTURE.md           # Arquitetura detalhada
    ├── MIGRATION_GUIDE.md        # Guia de migração
    └── QUICKSTART.md             # Início rápido
```

## 🚀 Início Rápido

### Anotação Programática (Nova!)

```python
from ritm_annotation.core.annotation import AnnotationSession
from ritm_annotation.inference.utils import load_is_model
from ritm_annotation.inference.predictors import get_predictor

# Carregar modelo
model = load_is_model("checkpoint.pth", device="cuda")
predictor = get_predictor(model, device="cuda")

# Criar sessão
session = AnnotationSession(predictor, prob_thresh=0.5)

# Anotar
session.load_image(image)
session.add_click(100, 150, is_positive=True)
session.add_click(200, 250, is_positive=True)
session.finish_object()

# Obter resultado
mask = session.get_result_mask()
```

### Treinamento Modular (Nova!)

```python
from ritm_annotation.core.training import (
    TrainingLoop, BatchProcessor, CheckpointManager
)

# Componentes modulares
batch_processor = BatchProcessor(model, loss_fn, metrics)
checkpoint_manager = CheckpointManager(checkpoint_dir)

loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    train_loader=train_loader,
    checkpoint_manager=checkpoint_manager,
)

# Treinar
history = loop.run(num_epochs=100)
```

### Web API (Nova!)

```bash
# Executar servidor
python examples/web_api_example.py

# Acessar docs
open http://localhost:8000/docs
```

## 🎯 Benefícios

### Antes (Monolítico)

```python
# ❌ Lógica acoplada à GUI
controller = InteractiveController(
    model, device,
    update_image_callback=self.update_image  # Depende de Tkinter
)

# ❌ Difícil testar sem GUI
# ❌ Não pode reusar em outros contextos
# ❌ Difícil adicionar nova interface
```

### Depois (Modular)

```python
# ✅ Lógica independente de UI
session = AnnotationSession(predictor)

# ✅ Fácil testar
assert len(session.state.get_current_object().clicks) == 1

# ✅ Reusar em qualquer contexto
adapter_gui = GUIAdapter(session)
adapter_web = WebAPIAdapter(session)

# ✅ Extensível via eventos
session.events.on(EventType.PREDICTION_COMPLETED, callback)
```

## 📊 Comparação

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Testabilidade** | Difícil (requer GUI) | Fácil (unit tests) |
| **Extensibilidade** | Acoplado | Eventos + Adapters |
| **Reutilização** | Baixa | Alta |
| **Interfaces** | Apenas Tkinter | Tkinter, Web, CLI, etc |
| **Manutenção** | Complexa | Simples (componentes focados) |
| **Documentação** | Espalhada | Centralizada + Exemplos |

## 🔄 Compatibilidade

### ✅ 100% Compatível

O código existente continua funcionando! A nova arquitetura usa adapters para manter compatibilidade:

```python
# GUI antiga ainda funciona
from ritm_annotation.cli.annotate import handle as annotate_command

# Mas agora você também pode usar o core diretamente
from ritm_annotation.core.annotation import AnnotationSession
```

### Migração Gradual

1. **Fase 1** (Atual): Componentes core + Adapters (código antigo funciona)
2. **Fase 2**: Migrar CLI commands para usar componentes novos
3. **Fase 3**: Adicionar interfaces web completas
4. **Fase 4**: Deprecar código legacy

## 📚 Documentação

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Arquitetura completa e design decisions
- **[MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)**: Como migrar código existente
- **[QUICKSTART.md](docs/QUICKSTART.md)**: Início rápido com exemplos

## 🎓 Exemplos de Uso

### 1. Processar Batch de Imagens

```python
for img_path in image_dir.glob("*.jpg"):
    session.load_image(load_image(img_path))

    # Aplicar clicks automáticos (de arquivo de config)
    for click in load_clicks(img_path):
        session.add_click(click.x, click.y, click.is_positive)

    # Salvar resultado
    mask = session.get_result_mask()
    save_mask(mask, output_dir / img_path.name)
```

### 2. Treinar com Early Stopping

```python
def on_epoch_end(epoch, metrics):
    if metrics.val_loss < best_loss:
        best_loss = metrics.val_loss
        patience = 0
    else:
        patience += 1
        if patience >= max_patience:
            return False  # Stop training
    return True

callbacks = {'on_epoch_end': on_epoch_end}
loop.run(num_epochs=1000, callbacks=callbacks)
```

### 3. API Web Personalizada

```python
@app.post("/annotate")
async def annotate_image(image: UploadFile, clicks: List[Click]):
    session = AnnotationSession(predictor)
    session.load_image(decode_image(image))

    for click in clicks:
        session.add_click(click.x, click.y, click.is_positive)

    return {"mask": encode_mask(session.get_result_mask())}
```

## 🧪 Testes

A nova arquitetura facilita testes unitários:

```python
def test_annotation_session():
    predictor = MockPredictor()
    session = AnnotationSession(predictor)

    session.load_image(test_image)
    session.add_click(100, 100, is_positive=True)

    assert len(session.state.get_current_object().clicks) == 1
    assert session.get_current_prediction() is not None
```

## 🎯 Casos de Uso Habilitados

### Antes (Limitado)

- ✅ GUI Desktop (Tkinter)
- ❌ Web interface
- ❌ CLI interativo
- ❌ Jupyter notebooks
- ❌ Batch processing
- ❌ Unit testing

### Depois (Flexível)

- ✅ GUI Desktop (Tkinter)
- ✅ Web interface (FastAPI, Flask, etc)
- ✅ CLI interativo
- ✅ Jupyter notebooks
- ✅ Batch processing
- ✅ Unit testing
- ✅ Mobile (futuro)
- ✅ Cloud services (futuro)

## 🔧 Componentes Principais

### AnnotationSession

```python
session = AnnotationSession(predictor, prob_thresh=0.5)
session.load_image(image)
session.add_click(x, y, is_positive=True)
session.undo_click()
session.finish_object()
result = session.get_result_mask()
```

### TrainingLoop

```python
loop = TrainingLoop(
    model, optimizer, batch_processor,
    train_loader, val_loader, checkpoint_manager
)
metrics = loop.run(num_epochs=100, callbacks=callbacks)
```

### GUIAdapter (Compatibilidade)

```python
# Wrapper que mantém interface antiga
adapter = GUIAdapter(session, update_image_callback=...)

# API idêntica ao InteractiveController antigo
adapter.set_image(image)
adapter.add_click(x, y, True)
adapter.undo_click()
```

## 🌟 Próximos Passos

1. **✅ Core implementado**: `core/annotation/` e `core/training/`
2. **✅ Adapters**: `GUIAdapter` para compatibilidade
3. **✅ Exemplos**: Web API básica
4. **✅ Documentação**: Arquitetura, migração, quickstart
5. **⏳ Testes**: Unit tests para componentes core
6. **⏳ Web UI**: Interface web completa
7. **⏳ CLI melhorado**: Usar novos componentes
8. **⏳ Performance**: Profiling e otimização

## 🤝 Contribuindo

Contribuições são muito bem-vindas! A arquitetura modular facilita:

1. **Adicionar interface**: Crie adapter em `interfaces/`
2. **Adicionar feature**: Estenda componentes em `core/`
3. **Melhorar docs**: Adicione exemplos em `examples/`
4. **Adicionar testes**: Teste componentes isoladamente

## 📄 Licença

[Mesma licença do projeto original]

## 🙏 Créditos

- **Arquitetura Original**: Equipe RITM
- **Refatoração Modular**: [Seu nome aqui]

---

## 💡 Filosofia

> "A melhor arquitetura é aquela que facilita mudanças, não aquela que prevê todas elas."

A nova arquitetura segue princípios SOLID:
- **S**ingle Responsibility: Cada componente tem uma responsabilidade
- **O**pen/Closed: Extensível via eventos e herança
- **L**iskov Substitution: Adapters são intercambiáveis
- **I**nterface Segregation: Interfaces pequenas e focadas
- **D**ependency Inversion: Core não depende de UI

---

**Para mais informações, consulte a [documentação completa](docs/ARCHITECTURE.md).**
