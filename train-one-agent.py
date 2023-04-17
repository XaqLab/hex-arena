from jarvis.config import Config, from_cli
from irc.manager import AgentManager

cli_args = Config({
    'store_dir': 'store',
    'defaults': 'irc-defaults.yaml',
    'env_param': [1, 0.001, 0.1, 0, 0.0286, 0.2, 0.0476, 0.2, 0.0667, 0.2],
})

if __name__=='__main__':
    cli_args.update(from_cli())
    manager = AgentManager(
        store_dir=cli_args.pop('store_dir'),
        defaults=cli_args.pop('defaults'),
        save_interval=3,
    )

    manager.train_one_agent(**cli_args)
