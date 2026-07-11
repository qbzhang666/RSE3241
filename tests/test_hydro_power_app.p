from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "hydro-power.py"


class HydroPowerAppTest(unittest.TestCase):
    def test_overview_to_step_one(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)

        app.run()
        self.assertEqual([], [item.value for item in app.exception])
        self.assertEqual(1, len(app.radio))

        navigation = app.radio[0]
        navigation.set_value(navigation.options[1]).run()
        self.assertEqual([], [item.value for item in app.exception])
        self.assertIn("Project setup", [item.value for item in app.subheader])
        self.assertGreaterEqual(len(app.number_input), 7)

        create_button = next(
            button for button in app.button if button.label == "Create new project"
        )
        create_button.click().run()
        self.assertEqual([], [item.value for item in app.exception])


if __name__ == "__main__":
    unittest.main()
