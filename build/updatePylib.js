/* eslint-disable @typescript-eslint/no-var-requires */
//@ts-check

const fse = require('fs-extra');
const path = require('path');

function copyToBin() {
    const base = path.join(__dirname, '..');
    const bin = path.join(base, 'bin');
    const pytea = path.join(base, 'packages', 'pytea');

    const pylibDest = path.join(bin, 'dist', 'pylib');
    if (fse.existsSync(pylibDest)) {
        fse.removeSync(pylibDest);
        console.log(`removed ${pylibDest}`);
    }
    fse.copySync(path.join(pytea, 'pylib'), pylibDest);
    console.log(`${path.join(pytea, 'pylib')} copied to ${pylibDest}`);
}

copyToBin();
