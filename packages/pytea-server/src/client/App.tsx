import axios from 'axios';
import * as React from 'react';
import Container from 'react-bootstrap/Container'
import Form from 'react-bootstrap/Form';

export interface IAppProps {}

export interface IAppState {
	name: string;
}
export default class App extends React.Component<IAppProps, IAppState> {
	state: IAppState = {
		name: 'default'
	}

	async componentDidMount() {
		try {
			const r = await axios.get('/api/version');
			const name: string = await r.data;
			this.setState({ name });
		} catch (error) {
			console.log(error);
		}
	}

	render() {
		return (
			<Container>
				<h3 className="text-primary text-center">PyTea analyzer</h3>

				<Form>
					<Form.File id="entry-path" label="click and select the python file" custom/>
				</Form>
			</Container>
		);
	}
}
