"""Compatible resume wrapper; original trainer and suite hashes remain unchanged."""
import train_revision as revision
from performance_runtime import install

original=revision.configure_model
def configure(base,method,rank,placement='mlp'):
    original(base,method,rank,placement)
    install(base,method)

if __name__=='__main__':
    revision.configure_model=configure
    revision.main()

