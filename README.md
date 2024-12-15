# RecTour2024Challenge

## Structure:
* data - All data-related files. Contains:
    * raw - The raw input to the project (Dataset from Kaggle).
    * processed - Data preprocessed.
* notebooks - Jupiter Notebooks.
* src - Source code. Contains:
    * data - Scripts for data loading.
    * models - Code for building, training, evaluating, and saving models.
    * utils - Utility scripts.

## Planning

- [ ] Preprocessing:
    - [ ] Find the input shape:
        - [ ] Find how to encode categorical for the NNs.
            - [ ] Guest country (Amount of countries for all sets combined).
            - [ ] Guest type (size = 4).
        - [ ] Month (Validate it's 1..12).
        - [ ] Amount of nights.
    - [ ] Check the output shape.

- [ ] Create NNs:
    - [ ] Find loss function.
    - [ ] Construct the NNs.
    - [ ] Train the NNs.

- [ ] Test performance:
    - [ ] Encode each categorical value for the test users.
    - [ ] Insert each test user to the NNs.
    - [ ] Use 0.333 coefficient for each section vector (Title, positive, negative) and find the combined vector.
    - [ ] Use similarity function to find the 10 closest reviews to each user.
    - [ ] Submit and check results.

- [ ] Find optimal coefficients for the combined vector.
    - [ ] Find loss function.
    - [ ] Use "Binary Search" method to find the optimal values.

### Other things to consider:
- [ ] Adding accommodations info as inputs.
- [ ] Ways to optimize the coefficients values.