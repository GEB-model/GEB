Create farmers
#############################

This file is used to create the farmer agents. If you have a map of all individual farms, you should use that. If not, then you can use :doc:`8_create_farmers` to create a statistically representative dataset of farmers. The first input `FARM_SIZE_CHOICES_M2` an array of the various farm sizes that exist in m\ :sup:`2`. Here a lower and upper bound are given for each choice of farm size.

The second input `FARM_SIZE_PROBABILITIES` is a map at the resolution and bounds of the submask, here placed in `ROOT/input/agents/farm_size/2010-11/farmsize.tif`, and as layers the probability that each of the previously defined farm sizes occurs in a given cell. Naturally, the probabilities for a given cell should add up to 1.

The function :func:`8_create_farmers.create_farmers` is then used to create the farmers on the cultivated land as exported in :doc:`5_create_land_use_map`. The function :func:`8_create_farmers.create_farms` which is used to create the actual farms, starts at the top-left going down row by row. When the scripts loops through the cells and hits a cell that should be cultivated, the script first checks what size the farm should be (using the specified farm size probabilities) and randomly choosing a farm size between the given bounds. Then the farm is randomly expanded to all sides, filling other cultivated cells until the given farm size is reached. The loop then continues until all cultivated land is filled by farms. Moreover, the center points (lon, lat) of all farms are collected in `farmer_locs`.

.. automodule:: preprocessing.8_create_farmers
    :members: