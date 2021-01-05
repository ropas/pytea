# How to build and debug

## build and run

```bash
npm run install:all
cd packages/pytea

npm run webpack # compile development build
# npm run build # compile production build

node ./index.js path/to/entry_python_script.py
```

## debug

VSCode의 debug 창에서 에서 Pytea CLI 또는 Pytea CLI scratch 구성을 선택해서 디버깅을 진행할 수 있다.

Pytea CLI를 선택시 pyteaconfig.json에서 `entryPath` 값을 python entry file path로 설정해주어야 디버깅을 진행할 수 있다.

Pytea CLI scratch는 자동으로 `pytest/basics/scratch.py`를 실행한다. Pytea CLI를 돌리면 기본적으로 webpack을 돌리기 때문에 typescript 컴파일 시간이 걸리지만 Pytea CLI scratch는 컴파일을 하지 않으므로 즉시 진행할 수 있다. (대신 컴파일이 미리 되어있어야한다.)

## when to compile

`npm run webpack`을 돌려서 컴파일시 `pytea/dist`에 컴파일한 js 파일과 pylib이 통째로 들어가게 된다.

pylib에서 .py 파일을 변경하였다면 반드시 컴파일을 해야 `node ./index.js ...`를 실행할 시 반영이 되므로 반드시 미리 `npm run webpack`을 돌려서 컴파일을 해주자.