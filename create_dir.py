import os

def create_project_structure(base_dir):
    structure = {
        "docs": {},
        "hardware": {
            "mechanical_design": {},
            "sensors": {},
            "controllers": {}
        },
        "software": {
            "vision": {},
            "speech": {},
            "motion_control": {},
            "integration": {}
        },
        "tests": {},
        "exhibits": {}
    }

    def create_dirs(base, structure):
        for folder, subfolders in structure.items():
            path = os.path.join(base, folder)
            os.makedirs(path, exist_ok=True)
            create_dirs(path, subfolders)

    create_dirs(base_dir, structure)

# Create the project structure in the current directory
project_name = "DualArm_Robot_Project"
current_dir = os.getcwd()
project_dir = os.path.join(current_dir, project_name)

if not os.path.exists(project_dir):
    os.makedirs(project_dir)

create_project_structure(project_dir)

print("Project structure for '{}' created successfully in {}.".format(project_name, current_dir))

