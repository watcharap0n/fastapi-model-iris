"""
response model api for iris
    :parameter
        - data -> dict
            - float -> float
            - float -> float
"""


from pydantic import BaseModel, conlist
from typing import List


class Iris(BaseModel):
    """
    :return
        - data type(list)
            - float
            - float
    """
    data: List[conlist(float, min_items=2, max_items=2)]
