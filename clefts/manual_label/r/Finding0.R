setwd(file.path(Mydirectories::googledrive.directory(),"Travail/Recherche/Travaux/Albert3"))
#0 Packages used
library("ggplot2") #for nice plots
library(ggfortify)#for graphs
library(plyr)
library(directlabels)

#1 Read and plot the data
data.connections=read.csv("all.csv")
data.connections<-data.connections[order(data.connections$contact_number),]
attach(data.connections)
N<-nrow(data.connections)
lmfit<-lm(synaptic_area~0+contact_number,weight=1/contact_number)
lmfit.l<-lm(synaptic_area~0+contact_number,weight=1/contact_number,data=data.connections[-c(24,68,71),])
table1<-data.frame(Circuit=circuit,
                   "Connections Count"=contact_number,
                   "Synaptic area"=synaptic_area)[ave(seq_along(circuit), circuit, FUN=seq_along)<=3,]
graph0<-ggplot(data=data.connections,aes(x=contact_number,y=synaptic_area,colour=circuit))+geom_point()+xlab("Number of contacts")+ylab("Synaptic Area in $\\mathrm{XX}^2$")

summary(lmfit)
graph1<-autoplot(lmfit)

summary(lmfit.l)
graph2<-autoplot(lmfit.l)

#checking formula:
X=contact_number;Y=synaptic_area;n=contact_number;Omega=diag(n)
solve(t(X)%*%solve(Omega)%*%X)%*%t(X)%*%solve(Omega)%*%Y

#checking qqplot
plot(sort(lmfit$residuals/(summary(lmfit)$sigma *sqrt(contact_number))),qnorm((1:nrow(data.connections))/(nrow(data.connections)+1)));abline(0,1)
ggplot(data=data.frame(stdresiduals=lmfit$residuals/(summary(lmfit)$sigma *sqrt(contact_number))),aes(x=stdresiduals))+geom_density()


#PI
sdpi.f<-function(newx,alpha){
  plyr::ddply(data.frame(alpha=alpha),"alpha",function(d){
  sdpi=sqrt(newx)*qt((1-d$alpha)/2,N-1)*(summary(lmfit)$sigma+coef(summary(lmfit))[,"Std. Error"])
  data.frame(newx=newx,y1=pmax(newx*lmfit$coefficients-sdpi,0),y2=newx*lmfit$coefficients+sdpi)})}

  newx=1:max(contact_number);pis=sdpi.f(newx,.95)

graph4<-ggplot(data=data.connections,aes(x=contact_number,y=synaptic_area))+geom_point()+
  geom_segment(data = pis,aes(x=newx,y=y1,xend=x,yend=y2))+
  xlab("Number of contacts")+ylab("Synaptic Area in $\\mathrm{XX}^2$")


graph5<-direct.label(ggplot(data=data.connections,aes(x=contact_number,y=synaptic_area))+geom_point()+
  geom_line(data=sdpi.f(newx,c(.9,.95,.99,.999,.999)),aes(x=newx,y=y1,group=alpha,colour=alpha))+
  geom_line(data=sdpi.f(newx,c(.9,.95,.99,.999,.999)),aes(x=newx,y=y2,group=alpha,colour=alpha))+
    xlab("Number of contacts")+ylab("Synaptic Area in $\\mathrm{XX}^2$"))
  

graph6<-direct.label(ggplot(data=data.connections[contact_number<18,],aes(x=contact_number,y=synaptic_area))+geom_point()+
                       geom_line(data=sdpi.f(1:18,c(.9,.95,.99,.999,.999)),aes(x=newx,y=y1,group=alpha,colour=alpha))+
                       geom_line(data=sdpi.f(1:18,c(.9,.95,.99,.999,.999)),aes(x=newx,y=y2,group=alpha,colour=alpha))+
                       xlab("Number of contacts")+ylab("Synaptic Area in $\\mathrm{XX}^2$"))

