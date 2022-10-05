### Mu, theta and solution amplitude ###########################################

library(data.table)
library(rgl)

# Extract the relevant tables

cwd <- getwd()

path_to_tables <- paste(dirname(cwd),'/', 'Tables', sep = '')

data_int <- fread(paste(path_to_tables,'/', 'tab_int.csv', sep = ''), header = FALSE)
data_bc  <- fread(paste(path_to_tables,'/', 'tab_bc.csv',  sep = ''), header = FALSE)

colnames(data_int) <- c('x','y','mu','theta','u')
colnames(data_bc)  <- c('x','y','mu','theta','u')

# Extract maximum and minimum per parameter

summary(data_bc$u)

max_u_int <- data_int[,max(u), by = mu]

plot(max_u_int$mu, max_u_int$V1, xlab = 'mu', ylab = 'u max', xaxt = 'n', ylim = c(0,max(max_u_int$V1)), main = 'Amplitude VS mu')
axis(side = 1, at = max_u_int$mu)


# Plot all the solutions for each mu (given theta = 0)

theta_0 = data_int$theta[1]
mus     = levels(factor(data_int$mu))

for (mui in mus)
{
  plot_points <- data_int[theta == theta_0 & mu == mui, .(x,y,u)]
  
  open3d()
  
  plot3d(plot_points$x,
         plot_points$y,
         plot_points$u,
         main = paste('mu =', mui))
}
