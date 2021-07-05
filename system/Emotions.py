import enum

# TODO may be better just to use string rather than enum since emotions we use keep changing
class Emotions(enum.Enum):
    NEUTRAL = 0
    HAPPINESS = 1
    SADNESS = 2
    FEAR = 3
    DISGUST = 4
    ANGER = 5
    SURPRISE = 6