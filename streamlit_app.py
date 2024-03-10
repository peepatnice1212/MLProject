import streamlit as st

def main():
    st.title('Welcome to My Streamlit Web App')
    
    # Get user input
    name = st.text_input('Enter your name:')
    color = st.selectbox('Select your favorite color:', ['Red', 'Green', 'Blue'])
    
    # Display greeting message
    if name:
        st.write(f'Hello, {name}!')
        st.write(f'Your favorite color is {color}.')
        st.write(f'Here is a {color.lower()} square:')
        st.write(f'<div style="width: 100px; height: 100px; background-color: {color};"></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
