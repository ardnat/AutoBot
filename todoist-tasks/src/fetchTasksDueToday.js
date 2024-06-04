const axios = require('axios');
const moment = require('moment');

const TODOIST_API_URL = 'https://api.todoist.com/rest/v1/tasks';
const TODOIST_API_TOKEN = 'your_todoist_api_token'; // Replace with your Todoist API token

async function fetchTasksDueToday() {
    try {
        const response = await axios.get(TODOIST_API_URL, {
            headers: {
                "Authorization": `Bearer ${TODOIST_API_TOKEN}`
            }
        });

        const tasks = response.data;

        const tasksDueToday = tasks.filter(task => {
            const dueDate = moment(task.due.date);
            return dueDate.isSame(moment(), 'day');
        });

        return tasksDueToday;
    } catch (error) {
        console.error('Error fetching tasks: ', error);
        throw error;
    }
}

module.exports = fetchTasksDueToday;