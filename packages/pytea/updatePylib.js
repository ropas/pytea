/* eslint-disable @typescript-eslint/no-var-requires */
//@ts-check

const fse = require('fs-extra');
const path = require('path');

function copyToBin() {
    const base = __dirname;
    const bin = path.join(base, 'dist');
    const pylibDest = path.join(bin, 'pylib');

    if (fse.existsSync(pylibDest)) {
        fse.removeSync(pylibDest);
        console.log(`removed ${pylibDest}`);
    }
    fse.copySync(path.join(base, 'pylib'), pylibDest);
    console.log(`${path.join(base, 'pylib')} copied to ${pylibDest}`);
}

copyToBin();
