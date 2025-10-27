import { Router } from 'express';
import RAGController from '../controllers/ragController';

const router = Router();
const ragController = new RAGController();

export const setRAGRoutes = (app) => {
    app.use('/api/rag', router);
    
    router.get('/:patientId/status', ragController.getRAGStatus.bind(ragController));
    router.post('/:patientId/assess', ragController.assessRAG.bind(ragController));
};