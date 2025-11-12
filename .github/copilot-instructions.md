when reviewing a pull request, please consider the following guidelines:

- Units should be included in variable names where applicable and preferably as SI units.
- All monetary units should be nominal USD (face value) for the respective years.
- All new or substantially edited functions have documentation in the style of documentation in other functions.
- Where neccesary, code comments are added focussing on *why* something is done, rather than *what* is done. The *what* should ideally be evident from the variable naming and structure.
- All added or substantially edited functions have type annotations for arguments and return types.
- Do not use .raster accessor from hydromt; use .rio accessor from rioxarray instead.