# Pull Request Checklist

## Documentation
- [ ] I have added documentation to new or substantially edited functions in the style of documentation in other functions.
- [ ] Where neccesary, code comments are added focussing on *why* something is done, rather than *what* is done. The *what* should ideally be evident from the variable naming and structure.
- [ ] All added or substantially edited functions have type annotations for arguments and return types.

## Clarity
- [ ] All variable names are clear and understandable by a non-domain expert.
- [ ] Units are included unless super widely used default and in SI units (for example all hydrology units are standard in m).
- [ ] All monetary units MUST be nominal USD (face value) for the respective years.

## Testing
- [ ] Tests that are not available on GitHub actions pass (primarly test_model.py).
- [ ] When there is an error, the code fails (early). I.e., the code doesn't silently fail.

*Thank you for helping maintain the code quality!*