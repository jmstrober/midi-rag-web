class Patient {
    id: string;
    name: string;
    age: number;
    medicalHistory: string[];

    constructor(id: string, name: string, age: number, medicalHistory: string[]) {
        this.id = id;
        this.name = name;
        this.age = age;
        this.medicalHistory = medicalHistory;
    }
}

export default Patient;