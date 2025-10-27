export class ClinicalProtocolService {
    private protocols: any[];

    constructor() {
        this.protocols = [];
    }

    public addProtocol(protocol: any): void {
        this.protocols.push(protocol);
    }

    public getProtocols(): any[] {
        return this.protocols;
    }

    public findProtocolById(id: string): any | undefined {
        return this.protocols.find(protocol => protocol.id === id);
    }

    public updateProtocol(id: string, updatedProtocol: any): boolean {
        const index = this.protocols.findIndex(protocol => protocol.id === id);
        if (index !== -1) {
            this.protocols[index] = { ...this.protocols[index], ...updatedProtocol };
            return true;
        }
        return false;
    }

    public deleteProtocol(id: string): boolean {
        const index = this.protocols.findIndex(protocol => protocol.id === id);
        if (index !== -1) {
            this.protocols.splice(index, 1);
            return true;
        }
        return false;
    }
}