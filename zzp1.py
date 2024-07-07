import os, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import click


@click.command()
@click.option('--name', '-n', default='World')
def cmd(name):
    print(name)
    print("BBB")

def main():
    cmd()

print("aaaa")

if __name__ == '__main__':
    main()