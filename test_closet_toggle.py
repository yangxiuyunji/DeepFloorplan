from utils.ocr import text_to_label, set_closet_enabled


def test_disable_closet_maps_to_background():
    set_closet_enabled(False)
    assert text_to_label('衣柜') == 0
    assert text_to_label('closet') == 0
    set_closet_enabled(True)
    assert text_to_label('衣柜') == 1
