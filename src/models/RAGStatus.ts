export class RAGStatus {
    patientId: string;
    status: 'Red' | 'Amber' | 'Green';
    assessmentDate: Date;

    constructor(patientId: string, status: 'Red' | 'Amber' | 'Green', assessmentDate: Date) {
        this.patientId = patientId;
        this.status = status;
        this.assessmentDate = assessmentDate;
    }
}