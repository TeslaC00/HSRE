# HSRE (Hairstyle Recommendation Engine)

## Description
HSRE (Hairstyle Recommendation Engine) is a system that suggests suitable hairstyles based on your face structure and hair type. It leverages the CelebA dataset to compare and recommend hairstyles used by celebrities, providing a personalized experience.

## Installation
To set up and run this project locally, follow these steps:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/TeslaC00/HSRE.git
   cd HSRE
   ```

2. **Set Up the Python Environment:**
   This project uses `uv` for dependency management. Ensure `uv` is installed, then run:
   ```sh
   uv sync
   ```

3. **Download the CelebA Dataset:**
   Follow the official instructions to download the CelebA dataset and place it in the appropriate directory.

## Git Workflow for Collaborators
To ensure a structured development process, follow these rules:

- **Do not work directly on the `main` branch** or merge into `main` without a pull request.
- **Create a new branch** for each new feature or improvement:
  ```sh
  git checkout -b <branch-name>
  ```

- **Push your branch to GitHub:**
  ```sh
  git push -u origin <branch-name>
  ```

- **Submit a pull request** to merge changes into `main`.

## Project Management
- This project is managed using `uv`.
- All project details, including dependencies and the Python version, can be found in `pyproject.toml`.

## Contributing
We welcome contributions! Please follow the Git workflow and ensure your changes are well-tested before submitting a pull request.

## License
This project is licensed under the MIT License.

---
