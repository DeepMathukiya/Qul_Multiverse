import unittest

from frontend.preview_logic import build_preview_items


class BuildPreviewItemsTests(unittest.TestCase):
    def test_pair_preview_has_two_items(self) -> None:
        pair = {
            "vertical_device_id": "phone-a",
            "horizontal_device_id": "phone-b",
            "vertical_image_b64": "a",
            "horizontal_image_b64": "b",
        }

        result = build_preview_items(pair=pair, devices=["phone-a", "phone-b"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Vertical camera · phone-a")
        self.assertEqual(result[1]["title"], "Horizontal camera · phone-b")

    def test_single_device_preview_uses_connected_device(self) -> None:
        result = build_preview_items(pair=None, devices=["phone-a"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Phone stream · phone-a")
        self.assertEqual(result[0]["device_id"], "phone-a")

    def test_empty_preview_when_no_devices(self) -> None:
        result = build_preview_items(pair=None, devices=[])

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
