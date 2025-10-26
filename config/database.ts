import { Sequelize } from 'sequelize';

const databaseConfig = {
    database: 'your_database_name',
    username: 'your_username',
    password: 'your_password',
    host: 'localhost',
    dialect: 'postgres', // or 'mysql', 'sqlite', etc.
};

const sequelize = new Sequelize(databaseConfig.database, databaseConfig.username, databaseConfig.password, {
    host: databaseConfig.host,
    dialect: databaseConfig.dialect,
});

const connectDatabase = async () => {
    try {
        await sequelize.authenticate();
        console.log('Connection to the database has been established successfully.');
    } catch (error) {
        console.error('Unable to connect to the database:', error);
    }
};

export { sequelize, connectDatabase };