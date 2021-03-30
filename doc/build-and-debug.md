# How to build and debug

## Build and run

```bash
npm run install:all
cd packages/pytea

npm run webpack # compile development build
# npm run build # compile production build

# run constraint generator only
node ./index.js path/to/script.py

# run with z3
# run `npm run webpack` before it. it requires compiled sources (./dist).
python pytea.py path/to/entry_python_script.py

```

```bash
cd packages/pytea
npm run webpack

# run frontend only (run without z3)
node index.js path/to/source.py --logLevel=reduced
```

We add two debug options from VSCode Debug panel.
- `Pytea CLI`: Build and run pytea frontend using `packages/pytea/pytest/basics/pyteaconfig.json`. It should define `entryPath` option.
- `Pytea CLI scratch`: Scratchpad debugging. Does not build pytea config, but analyze `packages/pytea/pytest/basics/scratch.py`. User should build pytea package before to run in.