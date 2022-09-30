from general_crystals import CrystalFactory

def test_descriptions():
    descriptions = CrystalFactory.get_descriptions()
    assert descriptions
