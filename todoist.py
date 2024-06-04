import datetime
import requests

def get_tasks_due_today(api_token):
    url = "https://api.todoist.com/rest/v2/tasks"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    params = {
        "filter": "(overdue | today) & (!assigned to: others)"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    tasks_due_today = []
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    for task in response.json():
        tasks_due_today.append(task)
    print(len(tasks_due_today))
    return tasks_due_today

def get_project_name_from_id(id):
    if id in projectCache:
        return projectCache[id]
    else:
        return "Unknown project"

projectCache={}

def get_projects(api_token):
    url = "https://api.todoist.com/rest/v2/projects"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    projects = []
    for project in response.json():
        projects.append(project)
    projectCache=projects

