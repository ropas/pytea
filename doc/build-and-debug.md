# How to build and debug

## Generate development build

```bash
npm run install:all

cd packages/pytea
npm run webpack # compile development build

# run constraint generator only
node ./index.js path/to/script.py

# run with z3
python pytea.py path/to/script.py
```

We added two debug options to VSCode Debug panel.
- `Pytea CLI`: Build and run pytea frontend using `packages/pytea/pytest/basics/pyteaconfig.json`. It should define `entryPath` option.
- `Pytea CLI scratch`: Scratchpad debugging. Does not build pytea config, but analyze `packages/pytea/pytest/basics/scratch.py`. User must build PyTea with `npm run build` before test it.
