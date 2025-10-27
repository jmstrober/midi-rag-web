import express from 'express';
import { setPatientRoutes } from './routes/patientRoutes';
import { setRAGRoutes } from './routes/ragRoutes';
import { authMiddleware } from './middleware/authMiddleware';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(authMiddleware);

setPatientRoutes(app);
setRAGRoutes(app);

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});