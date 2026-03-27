# Pull request

## Summary

- What changed?
- Why was it needed?

## Validation

- [ ] `python -m compileall scmetaintel geodh benchmarks scripts tests`
- [ ] `python -m unittest discover -s tests -p 'test_repository_health.py' -v`
- [ ] Relevant runtime or smoke checks were run when needed

## Checklist

- [ ] Documentation updated when behavior changed
- [ ] Placeholder/runtime directories remain intact
- [ ] No large local artifacts or secrets were added to git
