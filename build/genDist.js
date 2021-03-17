/* eslint-disable @typescript-eslint/no-var-requires */
//@ts-check

const fse = require('fs-extra');
const path = require('path');

function copyToBin() {
    const base = path.join(__dirname, '..');
    const bin = path.join(base, 'bin');
    const binDist = path.join(bin, 'dist');
    const pytea = path.join(base, 'packages', 'pytea');

    fse.mkdirSync(binDist, { recursive: true });

    fse.copySync(path.join(pytea, 'dist'), binDist);
    fse.moveSync(path.join(binDist, 'z3wrapper'), path.join(bin, 'z3wrapper'), { overwrite: true });

    fse.copyFileSync(path.join(pytea, 'index.js'), path.join(bin, 'index.js'));
    fse.copyFileSync(path.join(pytea, 'pytea.py'), path.join(bin, 'pytea.py'));
}

copyToBin();
