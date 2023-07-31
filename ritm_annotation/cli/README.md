This folder holds the subcommands related code

Each subcommand is defined in their own folder following the same pattern as the others

The only required thing is that the handle function must allow a `argparse.ArgumentParser` and return a function that receives the args
