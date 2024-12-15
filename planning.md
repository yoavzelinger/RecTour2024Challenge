# Planning
- Create the NNs.
    - Enter user info (encoding).
    - all enter to each NN (Title, Positive, Negative).
    - Train the NNs.

- Use the NNs:
    - Predict all the vectors.
    - find the coefficients for mixing the vectors:
        - for each coefficients:
            - Predict.
            - Calculate top10 for each user.
            - Calculate MRR@10.
        - Optimize.


## Inputs:
- Guest country (Categorical).
- GuestType (|Categorical| <= 4)
- [ ] Check how to transform to network valid input.
- Month (1.. i.. 12)
    - [ ] TODO - Validate it's starting with 1.
- days (Numeric).

## Output:
- [ ] Check about the input shape.
- [ ] Check about ways to create the review vectors.
    - Will give us the output shape.
- [ ] Decide about loss function.
- [ ] Thing of ways to get optimize coefficients.

## Optional to check:
[ ] Accommodations info as inputs. 
