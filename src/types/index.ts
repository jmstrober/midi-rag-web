export interface Patient {
    id: string;
    name: string;
    age: number;
    medicalHistory: string[];
}

export interface RAGStatus {
    patientId: string;
    status: 'Red' | 'Amber' | 'Green';
    assessmentDate: Date;
}