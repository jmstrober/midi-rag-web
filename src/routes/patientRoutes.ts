import { Router } from 'express';
import PatientController from '../controllers/patientController';

const router = Router();
const patientController = new PatientController();

export function setPatientRoutes(app) {
    app.use('/api/patients', router);
    router.get('/:id', patientController.getPatient.bind(patientController));
    router.post('/', patientController.createPatient.bind(patientController));
    router.put('/:id', patientController.updatePatient.bind(patientController));
}