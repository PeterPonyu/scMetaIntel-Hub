# Merge Notes

## Source projects

- `/home/zeyufu/Desktop/GEO-DataHub`
- `/home/zeyufu/Desktop/scMetaIntel`

## Adopted from GEO-DataHub

- broad model registry and benchmark-style configuration
- benchmark metric utilities
- practical model pull script style
- acquisition backbone and CLI

## Adopted from scMetaIntel

- dataclass-based study/query/result models
- chat REPL
- class-based answer / retrieval / ontology APIs
- richer user-facing architecture

## Key refinement decisions

1. **Keep both API styles where useful**
   - functional helpers for benchmarks
   - class wrappers for runtime UX

2. **Prefer runnable defaults over aspirational defaults**
   - recommended frontier models remain documented
   - default runtime models are small enough for immediate smoke tests

3. **Bridge before vendor**
   - GEO acquisition commands are exposed immediately
   - full copy-in migration can happen safely later
