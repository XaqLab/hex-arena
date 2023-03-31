from jarvis.config import Config, from_cli
from irc.manager import AgentManager

cli_args = Config({
    'store_dir': 'store',
    'defaults': 'irc-defaults.yaml',
    'param_grid': 'irc-grid.yaml',
})

if __name__=='__main__':
    cli_args.update(from_cli())
    manager = AgentManager(
        store_dir=cli_args.pop('store_dir'),
        defaults=cli_args.pop('defaults'),
    )
    manager.train_agents_on_grid(**cli_args)
