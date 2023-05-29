public class Client implements Observer {
    Integer id;
    public Client(Integer id){
        this.id = id;
    }

    @Override
    public void update(Subject subject, Object data) {
        System.out.println(id + " Received updated data: " + (Integer)data);
    }
    
}
