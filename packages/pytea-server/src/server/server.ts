import express from 'express';
import { runPyt } from 'pytea/main';

import apiRouter from './routes';

const app = express();

app.use(express.static('public'));
app.use(apiRouter);

const port = process.env.PORT || 3000;

function mainCallback() {
    // TODO
    console.log(`Server listening on port: ${port}`);
    runPyt;
}

app.listen(port, mainCallback);
