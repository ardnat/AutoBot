# Todoist Tasks Due Today

This project is a simple utility for fetching tasks that are due today from Todoist using the Todoist API.

## Setup

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run `npm install` to install the project dependencies.

## Usage

The main function of this project is `fetchTasksDueToday`, which is exported from the `src/fetchTasksDueToday.js` file.

Here is an example of how to use it:

```javascript
const fetchTasksDueToday = require('./src/fetchTasksDueToday');

fetchTasksDueToday()
  .then(tasks => console.log(tasks))
  .catch(err => console.error(err));
```

This will log an array of tasks due today to the console.

## Configuration

You will need to provide your Todoist API token. This can be found in the settings of your Todoist account. Once you have your token, you can set it as an environment variable named `TODOIST_API_TOKEN`.

## Running the Project

After setting up and configuring the project, you can run it with `node src/fetchTasksDueToday.js`.

## Dependencies

This project uses the following npm packages:

- `axios`: Used to make HTTP requests to the Todoist API.
- `moment`: Used to handle dates and determine if a task is due today.

## Contributing

If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request. Contributions are welcome and appreciated!