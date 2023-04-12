from jarvis.config import Config, from_cli
from irc.manager import AgentManager

cli_args = Config({
    'store_dir': 'store',
    'defaults': 'irc-defaults.yaml',
    'spec': 'distill-spec.yaml',
    'min_epoch': 12,
    'save_path': 'wukong.pt',
})

if __name__=='__main__':
    cli_args.update(from_cli())
    manager = AgentManager(
        store_dir=cli_args.pop('store_dir'),
        defaults=cli_args.pop('defaults'),
    )
    manager.distill_agents_from_spec(**cli_args)
