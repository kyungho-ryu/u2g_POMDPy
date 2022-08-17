from tensorboardX import SummaryWriter
writer = SummaryWriter()

writer.add_scalar("test/test", 3, 1)
writer.add_scalar("test/test2",4, 1)
writer.add_scalar("test/test3", 5, 1)

writer.add_scalar("test/test", 7, 2)
writer.add_scalar("test/test2",9, 2)
writer.add_scalar("test/test3", 8, 2)
