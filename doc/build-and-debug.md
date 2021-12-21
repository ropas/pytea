# How to build and debug

## Generate development build

```bash
npm run install:all

cd packages/pytea
npm run webpack # compile development build

# run constraint generator (online analyzer) only
node ./index.js path/to/script.py

# run full pytea
python pytea.py path/to/script.py
```

We use two debug options to VSCode Debug panel.
- `Pytea CLI`: Build and run pytea with `packages/pytea/pytest/basics/pyteaconfig.json`. The config file must have `entryPath` option.
- `Pytea CLI scratch`: Scratchpad debugging. It does not build pytea config, but analyzes `packages/pytea/pytest/basics/scratch.py`. User must build PyTea with `npm run build` before tests it.
