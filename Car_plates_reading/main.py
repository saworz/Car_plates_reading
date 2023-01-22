from features.choose_loading import choose_loading
from features.choose_mode import choose_mode

def main():
    model = choose_loading()
    choose_mode(model)

if __name__ == '__main__':
    main()