for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        real_score = discriminator(real_imgs)
        fake_score = discriminator(gen_imgs)

        real_score_sorted, _ = torch.sort(torch.reshape(real_score, (1, imgs.shape[0])))
        fake_score_sorted, _ = torch.sort(torch.reshape(fake_score, (1, imgs.shape[0])))

        g_loss = torch.mean((fake_score_sorted - real_score_sorted) ** 2)

        g_losses.append(g_loss)
        g_loss.backward()
        plot_grad_flow(generator.named_parameters())
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_score = discriminator(real_imgs)
        fake_score = discriminator(gen_imgs.detach())

        real_loss = adversarial_loss(real_score, valid)
        fake_loss = adversarial_loss(fake_score, fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)