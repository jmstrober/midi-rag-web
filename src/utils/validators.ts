export function validatePatientData(patient: any): boolean {
    if (!patient.name || typeof patient.name !== 'string') {
        return false;
    }
    if (!patient.age || typeof patient.age !== 'number' || patient.age < 0) {
        return false;
    }
    if (!Array.isArray(patient.medicalHistory)) {
        return false;
    }
    return true;
}

export function validateRAGStatus(ragStatus: any): boolean {
    const validStatuses = ['Red', 'Amber', 'Green'];
    if (!ragStatus.patientId || typeof ragStatus.patientId !== 'string') {
        return false;
    }
    if (!ragStatus.status || !validStatuses.includes(ragStatus.status)) {
        return false;
    }
    if (!ragStatus.assessmentDate || isNaN(new Date(ragStatus.assessmentDate).getTime())) {
        return false;
    }
    return true;
}