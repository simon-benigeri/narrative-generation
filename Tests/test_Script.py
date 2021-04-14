import unittest
from system.Script import Script


class TestScript(unittest.TestCase):

    arc = ['sad', 'happy']
    s = Script(arc)
    N = 3

    utterances = [f"utterance {i}" for i in range(N)]
    directions = [f"direction {i}" for i in range(N)]
    for u, d in zip(utterances, directions):
        s.append_dialogue(s.CHARACTER_X, [u])
        s.append_dialogue(s.CHARACTER_Y, [u])
        s.append_direction(s.CHARACTER_X, [d])
        s.append_direction(s.CHARACTER_Y, [d])

    def test_get_prev_utterances(self):
        self.assertEqual(len(self.s.get_prev_utterances(n=0, character=self.s.CHARACTER_X)), 0)
        self.assertEqual(len(self.s.get_prev_utterances(n=0, character=self.s.CHARACTER_Y)), 0)
        self.assertEqual(len(self.s.get_prev_utterances(n=2, character=self.s.CHARACTER_X)), 2)
        self.assertEqual(len(self.s.get_prev_utterances(n=2, character=self.s.CHARACTER_X)), 2)
        self.assertEqual(len(self.s.get_prev_utterances(n=self.N + 2, character=self.s.CHARACTER_X)), self.N)
        self.assertEqual(len(self.s.get_prev_utterances(n=self.N + 2, character=self.s.CHARACTER_X)), self.N)

    def test_get_prev_directions(self):
        self.assertEqual(len(self.s.get_prev_directions(n=0, character=self.s.CHARACTER_X)), 0)
        self.assertEqual(len(self.s.get_prev_directions(n=0, character=self.s.CHARACTER_Y)), 0)
        self.assertEqual(len(self.s.get_prev_directions(n=2, character=self.s.CHARACTER_X)), 2)
        self.assertEqual(len(self.s.get_prev_directions(n=2, character=self.s.CHARACTER_X)), 2)
        self.assertEqual(len(self.s.get_prev_directions(n=self.N + 2, character=self.s.CHARACTER_X)), self.N)
        self.assertEqual(len(self.s.get_prev_directions(n=self.N + 2, character=self.s.CHARACTER_X)), self.N)

    def test_get_prev_lines(self):
        self.assertEqual(len(self.s.get_prev_lines(n=0, type=self.s.UTTERANCE, character=self.s.CHARACTER_X)), 0)
        self.assertEqual(len(self.s.get_prev_lines(n=0, type=self.s.UTTERANCE, character=self.s.CHARACTER_Y)), 0)
        self.assertEqual(len(self.s.get_prev_lines(n=2, type=self.s.UTTERANCE, character=self.s.CHARACTER_X)), 2)
        self.assertEqual(len(self.s.get_prev_lines(n=2, type=self.s.UTTERANCE, character=self.s.CHARACTER_Y)), 2)
        self.assertEqual(len(self.s.get_prev_lines(n=self.N + 2, type=self.s.UTTERANCE, character=self.s.CHARACTER_X)),
                         self.N)
        self.assertEqual(len(self.s.get_prev_lines(n=self.N + 2, type=self.s.UTTERANCE, character=self.s.CHARACTER_Y)),
                         self.N)

        self.assertEqual(len(self.s.get_prev_lines(n=0, type=self.s.DIRECTION, character=self.s.CHARACTER_X)), 0)
        self.assertEqual(len(self.s.get_prev_lines(n=0, type=self.s.DIRECTION, character=self.s.CHARACTER_Y)), 0)
        self.assertEqual(len(self.s.get_prev_lines(n=2, type=self.s.DIRECTION, character=self.s.CHARACTER_X)), 2)
        self.assertEqual(len(self.s.get_prev_lines(n=2, type=self.s.DIRECTION, character=self.s.CHARACTER_Y)), 2)
        self.assertEqual(len(self.s.get_prev_lines(n=self.N + 2, type=self.s.DIRECTION, character=self.s.CHARACTER_X)),
                         self.N)
        self.assertEqual(len(self.s.get_prev_lines(n=self.N + 2, type=self.s.DIRECTION, character=self.s.CHARACTER_Y)),
                         self.N)

if __name__ == '__main__':
    unittest.main()
