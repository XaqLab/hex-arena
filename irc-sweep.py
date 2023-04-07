from jarvis.config import Config, from_cli
from irc.manager import AgentManager

cli_args = Config({
    'store_dir': 'store',
    'defaults': 'irc-defaults.yaml',
    'spec': 'sweep-spec.yaml',
    'count': 10,
})

if __name__=='__main__':
    cli_args.update(from_cli())
    manager = AgentManager(
        store_dir=cli_args.pop('store_dir'),
        defaults=cli_args.pop('defaults'),
        save_interval=3,
    )
    manager.train_agents_from_spec(**cli_args)
