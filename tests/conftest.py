import pytest

from molbart.modules.tokenizer import ChemformerTokenizer, SpanTokensMasker


@pytest.fixture
def example_tokens():
    return [
        ["^", "C", "(", "=", "O", ")", "unknown", "&"],
        ["^", "C", "C", "<SEP>", "C", "Br", "&"],
    ]


@pytest.fixture
def regex_tokens():
    regex = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    return regex.split("|")


@pytest.fixture
def smiles_data():
    return ["CCO.Ccc", "CCClCCl", "C(=O)CBr"]


@pytest.fixture
def mock_random_choice(mocker):
    class ToggleBool:
        def __init__(self):
            self.state = True

        def __call__(self, *args, **kwargs):
            states = []
            for _ in range(kwargs["k"]):
                states.append(self.state)
                self.state = not self.state
            return states

    mocker.patch("molbart.modules.tokenizer.random.choices", side_effect=ToggleBool())


@pytest.fixture
def setup_tokenizer(regex_tokens, smiles_data):
    def wrapper(tokens=None):
        return ChemformerTokenizer(
            smiles=smiles_data, tokens=tokens, regex_token_patterns=regex_tokens
        )

    return wrapper


@pytest.fixture
def setup_masker(setup_tokenizer):
    def wrapper(cls=SpanTokensMasker):
        tokenizer = setup_tokenizer()
        return tokenizer, cls(tokenizer)

    return wrapper
